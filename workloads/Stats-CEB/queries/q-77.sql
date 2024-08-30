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
  AND p.ViewCount >= 0
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 7
  AND p.FavoriteCount >= 0
  AND p.FavoriteCount <= 17
  AND v.VoteTypeId = 5
  AND b.Date >= CAST('2010-08-01 02:54:53' AS timestamp)
  AND u.Reputation >= 1
  AND u.Views >= 0
  AND u.CreationDate >= CAST('2010-08-19 06:26:34' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-11 05:22:26' AS timestamp);
