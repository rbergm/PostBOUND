SELECT COUNT(*)
FROM votes AS v,
  posts AS p,
  badges AS b,
  users AS u
WHERE u.Id = v.UserId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND p.Score >= 0
  AND p.Score <= 30
  AND p.CommentCount = 0
  AND p.CreationDate >= CAST('2010-07-27 15:30:31' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-04 17:45:10' AS timestamp);
