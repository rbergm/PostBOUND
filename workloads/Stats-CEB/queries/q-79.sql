SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND p.Id = c.PostId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND c.Score = 0
  AND p.Score <= 21
  AND p.AnswerCount <= 3
  AND p.FavoriteCount >= 0
  AND v.CreationDate >= CAST('2010-07-19 00:00:00' AS timestamp)
  AND b.Date <= CAST('2014-09-11 18:35:08' AS timestamp)
  AND u.Reputation <= 240;
