SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  votes AS v,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND u.Id = c.UserId
  AND u.Id = v.UserId
  AND c.CreationDate >= CAST('2010-07-27 12:03:40' AS timestamp)
  AND p.Score >= 0
  AND p.Score <= 28
  AND p.ViewCount >= 0
  AND p.ViewCount <= 6517
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 5
  AND p.FavoriteCount >= 0
  AND p.FavoriteCount <= 8
  AND p.CreationDate >= CAST('2010-07-27 11:29:20' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-13 02:50:15' AS timestamp)
  AND u.Views >= 0
  AND u.CreationDate >= CAST('2010-07-27 09:38:05' AS timestamp);
