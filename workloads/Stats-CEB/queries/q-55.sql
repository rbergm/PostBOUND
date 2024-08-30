SELECT COUNT(*)
FROM tags AS t,
  posts AS p,
  users AS u,
  postHistory AS ph,
  badges AS b
WHERE p.Id = t.ExcerptPostId
  AND u.Id = ph.UserId
  AND u.Id = b.UserId
  AND u.Id = p.OwnerUserId
  AND p.CommentCount >= 0
  AND u.DownVotes <= 0
  AND b.Date <= CAST('2014-08-22 02:21:55' AS timestamp);
